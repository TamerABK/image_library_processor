pragma ComponentBehavior: Bound

import QtQuick
import "../layout" as Room36Layout
import "../theme" as Room36Theme
import "../dev" as Dev
import "../components" as Components
import "../navigation/LibraryRoutes.js" as Routes

Rectangle {
    id: root
    objectName: "libraryShell"

    property var projectId: null
    property var photoModel: null
    property var processingController: null
    property var peopleController: null
    property var inspectorController: null
    property string libraryViewSize: "Medium"
    property string blurViewSize: "Medium"
    property string knownPeopleViewSize: "Medium"
    property string unknownPeopleViewSize: "Medium"
    property string projectName: ""
    property string currentRoute: "library"
    property var routeCounts: ({
        blurry: processingController ? processingController.routeCounts.blurry : -1,
        knownPeople: peopleController ? peopleController.knownCount : -1,
        unknownPeople: peopleController ? peopleController.unknownCount : -1
    })
    readonly property var currentEntry: Routes.entry(currentRoute)
    readonly property var routeStage: processingController && processingController.stages.length > 0
        ? processingController.stages[currentRoute === "duplicates" ? 1 : 5] : ({state: "NOT_STARTED"})

    signal homeRequested()
    signal settingsRequested()
    signal routeSelected(string routeId)
    signal sortRequested(string routeId, string option)
    signal viewSizeRequested(string routeId, string option)

    Room36Theme.Theme { id: theme }
    Room36Theme.Metrics { id: metrics }

    color: theme.libraryBackground
    implicitWidth: metrics.minimumWindowWidth
    implicitHeight: metrics.minimumWindowHeight

    function showRoute(routeId) {
        if (inspectorController) inspectorController.close()
        currentRoute = Routes.normalize(routeId)
    }

    onCurrentRouteChanged: {
        const normalized = Routes.normalize(currentRoute)
        if (currentRoute !== normalized) {
            currentRoute = normalized
            return
        }
        routeSelected(currentRoute)
    }

    Room36Layout.LibrarySidebar {
        id: sidebar
        anchors.left: parent.left
        anchors.top: parent.top
        anchors.bottom: parent.bottom
        width: root.width >= metrics.libraryWideBreakpoint
            ? metrics.sidebarPreferredWidth : metrics.sidebarCompactWidth
        currentRoute: root.currentRoute
        routeCounts: root.routeCounts
        onRouteRequested: function(routeId) { root.showRoute(routeId) }
        onHomeRequested: root.homeRequested()
        onSettingsRequested: root.settingsRequested()
    }

    Room36Layout.LibraryTopBar {
        id: topBar
        anchors.left: sidebar.right
        anchors.right: parent.right
        anchors.top: parent.top
        height: implicitHeight
        projectName: root.projectName
        pageTitle: root.currentEntry.title
        libraryControlsEnabled: ["library", "blurry", "knownPeople", "unknownPeople"].indexOf(root.currentRoute) >= 0
        sortOptions: root.currentRoute === "blurry"
            ? ["Most problematic first", "Least problematic first", "Filename A–Z", "Filename Z–A"]
            : root.currentRoute === "knownPeople"
                ? ["Name A–Z", "Most photos", "Most faces"]
                : root.currentRoute === "unknownPeople"
                    ? ["Largest clusters", "Smallest clusters"]
                    : ["Name A–Z", "Name Z–A"]
        onHomeRequested: root.homeRequested()
        onSortRequested: function(option) {
            if (root.currentRoute === "library" && root.photoModel && ["Name A–Z", "Name Z–A"].indexOf(option) >= 0)
                root.photoModel.sortByFilename(option === "Name Z–A")
            else if (root.currentRoute === "blurry" && root.processingController)
                root.processingController.blurModel.setSort(option)
            else if (root.currentRoute === "knownPeople" && root.peopleController)
                root.peopleController.knownModel.setSort(option)
            else if (root.currentRoute === "unknownPeople" && root.peopleController)
                root.peopleController.unknownModel.setSort(option)
            root.sortRequested(root.currentRoute, option)
        }
        onViewSizeRequested: function(option) {
            if (["Small", "Medium", "Large"].indexOf(option) >= 0) {
                if (root.currentRoute === "library") root.libraryViewSize = option
                else if (root.currentRoute === "blurry") root.blurViewSize = option
                else if (root.currentRoute === "knownPeople") root.knownPeopleViewSize = option
                else if (root.currentRoute === "unknownPeople") root.unknownPeopleViewSize = option
            }
            root.viewSizeRequested(root.currentRoute, option)
        }
    }

    Loader {
        id: contentLoader
        objectName: "libraryContentLoader"
        anchors.left: sidebar.right
        anchors.right: parent.right
        anchors.top: processingStatus.bottom
        anchors.bottom: parent.bottom
        anchors.margins: metrics.libraryContentMargin

        sourceComponent: root.currentRoute === "library" ? gridComponent
            : root.currentRoute === "blurry" && root.processingController ? blurComponent
            : root.currentRoute === "knownPeople" ? knownComponent
            : root.currentRoute === "unknownPeople" ? unknownComponent : placeholderComponent
    }

    Components.ProcessingStatus {
        id: processingStatus
        anchors.left: sidebar.right
        anchors.right: parent.right
        anchors.top: topBar.bottom
        controller: root.processingController
    }

    Component {
        id: blurComponent
        BlurryPhotosPage {
            processing: root.processingController
            inspector: root.inspectorController
            viewSize: root.blurViewSize
        }
    }

    Component {
        id: knownComponent
        KnownPeoplePage {
            people: root.peopleController
            inspector: root.inspectorController
            viewSize: root.knownPeopleViewSize
        }
    }

    Component {
        id: unknownComponent
        UnknownPeoplePage {
            people: root.peopleController
            inspector: root.inspectorController
            viewSize: root.unknownPeopleViewSize
        }
    }

    Component {
        id: gridComponent
        Components.PhotoGrid {
            photoModel: root.photoModel
            viewSize: root.libraryViewSize
        }
    }

    Component {
        id: placeholderComponent
        Dev.LibraryRoutePlaceholder {
            routeId: root.currentRoute
            title: root.currentEntry.title
            message: root.currentEntry.message + (root.processingController && ["duplicates", "knownPeople", "unknownPeople"].indexOf(root.currentRoute) >= 0
                ? "\n" + (root.routeStage.state === "COMPLETE"
                    ? "Analysis is available. Review tools will arrive in a later phase."
                    : "Analysis status: " + root.routeStage.state)
                : "")
        }
    }
}
