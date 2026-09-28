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
    property string libraryViewSize: "Medium"
    property string projectName: ""
    property string currentRoute: "library"
    property var routeCounts: ({})
    readonly property var currentEntry: Routes.entry(currentRoute)

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
        onHomeRequested: root.homeRequested()
        onSortRequested: function(option) { root.sortRequested(root.currentRoute, option) }
        onViewSizeRequested: function(option) {
            if (root.currentRoute === "library" && ["Small", "Medium", "Large"].indexOf(option) >= 0)
                root.libraryViewSize = option
            root.viewSizeRequested(root.currentRoute, option)
        }
    }

    Loader {
        id: contentLoader
        objectName: "libraryContentLoader"
        anchors.left: sidebar.right
        anchors.right: parent.right
        anchors.top: topBar.bottom
        anchors.bottom: parent.bottom
        anchors.margins: metrics.libraryContentMargin

        sourceComponent: root.currentRoute === "library" ? gridComponent : placeholderComponent
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
            message: root.currentEntry.message
        }
    }
}
